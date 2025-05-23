// SPDX-FileCopyrightText: Copyright (c) 2025 Schelling Point Ventures Inc.
// SPDX-License-Identifier: MIT
//
#include "Server.hpp"

#include <llama/Model.hpp>
#include <llama/Instance.hpp>
#include <llama/Session.hpp>
#include <llama/LogitComparer.hpp>

#include <bstl/thread_runner.hpp>
#include <bstl/move_capture.hpp>

#include <boost/asio/io_context.hpp>
#include <boost/asio/executor_work_guard.hpp>
#include <boost/asio/post.hpp>

namespace asio = boost::asio;

namespace bl::llama::server {

struct Server::Impl {
    std::shared_ptr<Model> m_model;
    bl::llama::Instance m_instance;

    asio::io_context m_ioctx;
    asio::executor_work_guard<asio::io_context::executor_type> m_wg;

    bstl::thread_runner m_runner;

    Impl(std::shared_ptr<Model> model)
        : m_model(std::move(model))
        , m_instance(*m_model, {})
        , m_wg(make_work_guard(m_ioctx))
        , m_runner(m_ioctx, 1)
    {
        m_instance.warmup();
    }

    ~Impl() {
        m_wg.reset();
    }

    void completeText(CompleteRequestParams params, itlib::ufunction<void(CompleteReponse)> cb) {
        post(m_ioctx, [this, movecap(params, cb)] {
            auto& session = m_instance.startSession({
                .seed = params.seed,
                .temperature = params.temperature,
                .topP = params.topP
                });
            session.setInitialPrompt(m_model->vocab().tokenize(params.prompt, true, true));
            auto iRes = session.complete({
                .prompt = m_model->vocab().tokenize(params.prompt, true, true),
                .maxTokens = (int32_t)params.maxTokens
                });

            CompleteReponse response;
            response.reserve(iRes.size());
            for (const auto& token : iRes) {
                auto& tokenData = response.emplace_back();
                tokenData.tokenStr = m_model->vocab().tokenToString(token.token);
                tokenData.tokenId = token.token;
                tokenData.logits.reserve(token.logits.size());
                for (const auto& logit : token.logits) {
                    TokenData::LogitData logitData;
                    logitData.tokenId = logit.token;
                    logitData.logit = logit.logit;
                    tokenData.logits.push_back({ (uint32_t)logit.token, logit.logit });
                }
            }

            cb(std::move(response));

            m_instance.stopSession();
        });
    }

    void verify(CompleteRequestParams req, CompleteReponse resp, itlib::ufunction<void(float)> cb) {
        post(m_ioctx, [this, movecap(req, resp, cb)] {
            auto& session = m_instance.startSession({
                .seed = req.seed,
                .temperature = req.temperature,
                .topP = req.topP
                });

            session.setInitialPrompt(m_model->vocab().tokenize(req.prompt, true, true));
            std::vector<TokenPrediction> origPredictions;
            origPredictions.reserve(resp.size());
            for (const auto& token : resp) {
                auto& tokenPrediction = origPredictions.emplace_back();
                tokenPrediction.token = token.tokenId;
                tokenPrediction.logits.reserve(token.logits.size());
                for (const auto& logit : token.logits) {
                    TokenData::LogitData logitData;
                    logitData.tokenId = logit.tokenId;
                    logitData.logit = logit.logit;
                    tokenPrediction.logits.push_back({ (int32_t)logit.tokenId, logit.logit });
                }
            }
            auto verifierPredictions = session.fillCtx(origPredictions);

            bl::llama::MetricsAggregator metricsAgg;
            float score = 0;
            for (size_t i = 0; i < origPredictions.size(); i++) {
                auto m = bl::llama::LogitComparer::compare(origPredictions[i].logits, verifierPredictions[i].logits);
                score = metricsAgg.pushAndVerify({ &m, 1 });
            }
            cb(score);

            m_instance.stopSession();
        });
    }
};

Server::Server(std::shared_ptr<Model> model)
    : m_impl(std::make_unique<Impl>(std::move(model)))
{
}

void Server::completeText(CompleteRequestParams params, itlib::ufunction<void(CompleteReponse)> cb) {
    m_impl->completeText(std::move(params), std::move(cb));
}

void Server::verify(CompleteRequestParams req, CompleteReponse resp, itlib::ufunction<void(float)> cb) {
    m_impl->verify(std::move(req), std::move(resp), std::move(cb));
}

Server::~Server() = default;

} // namespace bl::llama::server
