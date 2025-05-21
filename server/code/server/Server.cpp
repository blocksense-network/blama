// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#include "Server.hpp"
#include <llama/Model.hpp>
#include <llama/Instance.hpp>
#include <llama/Session.hpp>
#include <llama/LogitComparer.hpp>

namespace bl::llama::server {

struct Server::Impl {
    std::shared_ptr<Model> m_model;
    Impl(std::shared_ptr<Model> model)
        : m_model(std::move(model))
    {}
};

Server::Server(std::shared_ptr<Model> model)
    : m_impl(std::make_unique<Impl>(std::move(model)))
{}

void Server::completeText(CompleteRequestParams params, itlib::ufunction<void(std::vector<TokenData>)> cb) {
    bl::llama::Instance instance(*m_impl->m_model, {});
    instance.warmup();
    auto& session = instance.startSession({
        .seed = params.seed,
        .temperature = params.temperature,
        .topP = params.topP
    });
    session.setInitialPrompt(m_impl->m_model->vocab().tokenize(params.prompt, true, true));
    auto iRes = session.complete({
        .prompt = m_impl->m_model->vocab().tokenize(params.prompt, true, true),
        .maxTokens = (int32_t)params.maxTokens
    });

    std::vector<TokenData> response;
    response.reserve(iRes.size());
    for (const auto& token : iRes) {
        TokenData tokenData;
        tokenData.tokenStr = m_impl->m_model->vocab().tokenToString(token.token);
        tokenData.tokenId = token.token;
        tokenData.logits.reserve(token.logits.size());
        for (const auto& logit : token.logits) {
            TokenData::LogitData logitData;
            logitData.tokenId = logit.token;
            logitData.logit = logit.logit;
            tokenData.logits.push_back({ (uint32_t)logit.token, logit.logit });
        }
        response.push_back(tokenData);
    }

    cb(response);

    instance.stopSession();
}

void Server::verify(CompleteRequestParams req, std::vector<TokenData> resp, itlib::ufunction<void(float)> cb) {
        bl::llama::Instance instance(*m_impl->m_model, {});
    instance.warmup();
    auto& session = instance.startSession({
        .seed = req.seed,
        .temperature = req.temperature,
        .topP = req.topP
    });

    session.setInitialPrompt(m_impl->m_model->vocab().tokenize(req.prompt, true, true));
    std::vector<TokenPrediction> origPredictions;
    origPredictions.reserve(resp.size());
    for (const auto& token : resp) {
        TokenPrediction tokenPrediction;
        tokenPrediction.token = token.tokenId;
        tokenPrediction.logits.reserve(token.logits.size());
        for (const auto& logit : token.logits) {
            TokenData::LogitData logitData;
            logitData.tokenId = logit.tokenId;
            logitData.logit = logit.logit;
            tokenPrediction.logits.push_back({ (int32_t)logit.tokenId, logit.logit });
        }
        origPredictions.push_back(tokenPrediction);
    }
    auto verifierPredictions = session.fillCtx(origPredictions);

    bl::llama::MetricsAggregator metricsAgg;
    float score = 0;
    for (size_t i = 0; i < origPredictions.size(); i++) {
        auto m = bl::llama::LogitComparer::compare(origPredictions[i].logits, verifierPredictions[i].logits);
        score = metricsAgg.pushAndVerify({ &m, 1 });
    }
    cb(score);
}

Server::~Server() = default;

} // namespace bl::llama::server
