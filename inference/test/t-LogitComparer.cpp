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
    bl::llama::MetricsAggregator metricsAgg;
    auto score = metricsAgg.pushAndVerify({&metrics, 1});
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

    std::vector<bl::llama::TokenPrediction> iRes = session.complete({
        .maxTokens = 20
    });

    // compare with the same model
    bl::llama::Model model2(Model_117m_q6_k, {
        .gpu = false
    });
    bl::llama::Instance instance2(model2, {});
    auto& session2 = instance2.startSession({});
    session2.setInitialPrompt(model.vocab().tokenize("President George W.", true, true));
    auto iRes2 = session2.fillCtx(iRes);

    bl::llama::MetricsAggregator metricsAgg;
    float sumSim = 0;
    float score = 0;
    for (size_t i = 0; i < iRes.size(); i++) {
        float sim = bl::llama::LogitComparer::logitSimilarity(iRes[i].logits, iRes2[i].logits);
        auto m = bl::llama::LogitComparer::compare(iRes[i].logits, iRes2[i].logits);
        score = metricsAgg.pushAndVerify({&m, 1});
        sumSim += sim;
    }

    float averageSim = sumSim / iRes.size();
    CHECK(averageSim >= 0.98f);

    CHECK(score >= 0.95f);
}
