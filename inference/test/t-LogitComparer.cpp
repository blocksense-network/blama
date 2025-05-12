// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#include <doctest/doctest.h>

#include <llama/LogitComparer.hpp>
#include <llama/Model.hpp>
#include <llama/Instance.hpp>
#include <llama/Session.hpp>

#include "ac-test-data-llama-dir.h"

TEST_CASE("compare - no model") {
    bl::llama::TokenDataVector tdv1;
    bl::llama::TokenDataVector tdv2;

    float logitValue = 17.5f;
    for (int32_t i = 0; i < 10; i++){
        tdv1.push_back({i, logitValue});
        tdv2.push_back({i, logitValue});

        logitValue -= 0.5f;
    }

    // weighted similarity check
    auto res = bl::llama::LogitComparer::logitSimilarity(tdv1, tdv2);
    CHECK(res == 1.0f);

    // more complex compare func
    auto metrics = bl::llama::LogitComparer::compare(tdv1, tdv2);
    CHECK(metrics.top1Match == 1.0f);
    CHECK(metrics.distance == 0.0f);
    CHECK(metrics.jsd == 0.0f);

    // evalueate metrics
    auto score = bl::llama::LogitComparer::comparisonFinalScore({&metrics, 1});
    CHECK(score == 1.0f);
}

TEST_CASE("compare - with model") {
    const char* Model_117m_q6_k = AC_TEST_DATA_LLAMA_DIR "/gpt2-117m-q6_k.gguf";
    bl::llama::Model model(Model_117m_q6_k, {});

    // create inference instance
    bl::llama::Instance instance(model, {});

    // start session
    auto& session = instance.startSession({});
    session.setInitialPrompt(model.vocab().tokenize("President George W.", true, true));

    std::vector<bl::llama::TokenPrediction> iRes;

    constexpr int maxTokens = 20;
    for (int i = 0; i < maxTokens; ++i) {
        auto pred = session.getToken();
        if (pred.token == bl::llama::Token_Invalid) {
            // no more tokens
            break;
        }
        iRes.push_back(pred);
    }

    // compare with the same model
    bl::llama::Model model2(Model_117m_q6_k, {
        .gpu = false
    });
    bl::llama::Instance instance2(model2, {});
    auto& session2 = instance2.startSession({});
    session2.setInitialPrompt(model.vocab().tokenize("President George W.", true, true));
    auto iRes2 = session2.fillCtx(iRes);

    std::vector<bl::llama::LogitComparer::ComparisonMetrics> metrics(iRes.size());
    float sumSim = 0;
    for (size_t i = 0; i < iRes.size(); i++) {
        float sim = bl::llama::LogitComparer::logitSimilarity(iRes[i].logits, iRes2[i].logits);
        metrics[i] = bl::llama::LogitComparer::compare(iRes[i].logits, iRes2[i].logits);
        sumSim += sim;
    }

    float averageSim = sumSim / iRes.size();
    CHECK(averageSim >= 0.98f);

    float finalScore = bl::llama::LogitComparer::comparisonFinalScore(metrics);
    CHECK(finalScore >= 0.95f);
}
