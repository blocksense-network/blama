// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//

// llama
#include <llama/Init.hpp>
#include <llama/Model.hpp>
#include <llama/Instance.hpp>
#include <llama/Session.hpp>
#include <llama/LogitComparer.hpp>
#include <llama/ChatFormat.hpp>

// logging
#include <jalog/Instance.hpp>
#include <jalog/sinks/DefaultSink.hpp>

// model source directory
#include "ac-test-data-llama-dir.h"

#include <iostream>
#include <string>

#define SIMULATE_REST_REQUEST_API

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
    bl::llama::Instance instance(model, {});

    bl::llama::Model modelCpu(modelGguf, { .gpu = false}, modelLoadProgressCallback);
    bl::llama::Instance instanceCpu(modelCpu, {});

    auto roleUser = "User";
    auto roleAssistant = "Assistant";

    std::vector<bl::llama::ChatMsg> messages;
    std::string systemPrompt = "The first person to";
    messages.push_back({ "system", systemPrompt });

#if !defined(SIMULATE_REST_REQUEST_API)
    // start session
    auto& session = instance.startSession({});
    auto& sessionCpu = instanceCpu.startSession({});

    session.setInitialPrompt(model.vocab().tokenize(systemPrompt, true, true));;
    sessionCpu.setInitialPrompt(modelCpu.vocab().tokenize(systemPrompt, true, true));
#endif

    bl::llama::ChatFormat::Params chatParams = bl::llama::ChatFormat::getChatParams(model);
    if (chatParams.chatTemplate.empty()) {
        // GPT2 compatible template
        chatParams.chatTemplate = "{% for message in messages %}{{ message.role }} {{ message.content }}{{ eos_token }}{% endfor %}";

        // Other example for a template. That includes check for assistant prompt:
        // chatParams.chatTemplate =
                        // "{% for message in messages %}"
                        // "{{ '<|' + message['role'] + '|>\\n' + message['content'] + '<|end|>' + '\\n' }}"
                        // "{% endfor %}"
                        // "{% if add_generation_prompt %}"
                        // "{{ '<|' + assistant_role + '|>\\n' }}"
                        // "{% endif %}";
    }
    chatParams.roleAssistant = roleAssistant;
    bl::llama::ChatFormat chatFormat(chatParams);

    while (true) {
        std::cout << roleUser <<": ";
        std::string userPrompt;
        while (userPrompt.empty()) {
            std::getline(std::cin, userPrompt);
        }
        if (userPrompt == "/quit") break;

        auto userMsg = bl::llama::ChatMsg({ roleUser, userPrompt });
        messages.push_back(userMsg);

        std::vector<bl::llama::Token> tokenizedPrompt;

#if defined(SIMULATE_REST_REQUEST_API)
        auto formattedChat = chatFormat.formatChat(messages, true);

        auto& session = instance.startSession({});
        auto& sessionCpu = instanceCpu.startSession({});

        session.setInitialPrompt(model.vocab().tokenize(formattedChat, true, true));
        sessionCpu.setInitialPrompt(modelCpu.vocab().tokenize(formattedChat, true, true));
#else
        auto formatted = chatFormat.formatMsg(userMsg, messages, true);
        tokenizedPrompt = model.vocab().tokenize(formatted, true, true);
        sessionCpu.pushPrompt(modelCpu.vocab().tokenize(formatted, true, true));
#endif

        auto iRes = session.complete({
            .prompt = tokenizedPrompt,
            .maxTokens = 100
        });

        std::string response;
        for (const auto& token : iRes) {
            response += model.vocab().tokenToString(token.token);;
        }

        std::cout << roleAssistant << ": " << response << "\n";
        messages.push_back({ roleAssistant, response });

        auto iRes2 = sessionCpu.fillCtx(iRes);

        std::vector<bl::llama::LogitComparer::ComparisonMetrics> metrics(iRes.size());
        float sumSim = 0;
        for (size_t i = 0; i < iRes.size(); i++) {
            float sim = bl::llama::LogitComparer::logitSimilarity(iRes[i].logits, iRes2[i].logits);
            metrics[i] = bl::llama::LogitComparer::compare(iRes[i].logits, iRes2[i].logits);
            sumSim += sim;
        }

        std::cout << "\n\nAverage similarity: " << sumSim / iRes.size() << "\n";
        std::cout << "Final metrics score: " << bl::llama::LogitComparer::comparisonFinalScore(metrics) << "\n";

        std::cout << "\n";

#if defined(SIMULATE_REST_REQUEST_API)
        instance.stopSession();
        instanceCpu.stopSession();
#endif
    }

    return 0;
}
catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
}
